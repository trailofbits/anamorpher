# Contributing to Anamorpher

## Development Setup

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run linter and formatter
uv run ruff check --fix
uv run ruff format

# Run type checker
uv run ty check
```

## Dependency Compatibility Matrix

This project depends on multiple ML libraries with **fragile interdependencies**. When updating dependencies, you must validate compatibility across the entire stack.

### Critical Constraint: NumPy < 2.0

| Constraint | Reason | Symptom if Violated |
|------------|--------|---------------------|
| `numpy>=1.24.0,<2.0.0` | TensorFlow and OpenCV wheels are compiled against NumPy 1.x ABI | `AttributeError: _ARRAY_API not found` at import time |

**Why this matters:** TensorFlow and OpenCV distribute pre-compiled binary wheels built against NumPy 1.x. These wheels contain C extensions that are ABI-incompatible with NumPy 2.x. When NumPy 2.x is installed, imports fail immediately with cryptic `_ARRAY_API` errors.

### Package Compatibility

| Package | Constraint | NumPy 2.x Compatible | Notes |
|---------|------------|----------------------|-------|
| tensorflow | `>=2.15.0` | No | No NumPy 2.x wheels published yet |
| opencv-python | `>=4.8.0` | No | Binaries compiled against NumPy 1.x |
| torch | `>=2.0.0` | Yes | PyTorch wheels are forward-compatible |
| Pillow | `>=10.0.0` | Yes | Pure Python + C extension rebuilt for 2.x |

### Update Policy

When updating ML dependencies (tensorflow, torch, opencv-python, numpy, Pillow):

1. Run `uv sync` locally
2. Run `uv run pytest` - watch for `_ARRAY_API` errors
3. Test on **both** Python 3.11 and 3.12 (CI runs both)
4. Do NOT merge if CI fails with import errors

## Pull Request Checklist

### For ML Library Updates

- [ ] CI passes on **both** Python 3.11 and 3.12
- [ ] No `_ARRAY_API not found` errors in test output
- [ ] NumPy remains pinned to `<2.0.0` (until TF/OpenCV publish 2.x-compatible wheels)
- [ ] Reviewed this file's compatibility matrix

### For Web Framework Updates (Flask, bleach, etc.)

- [ ] CI passes
- [ ] Python version support matches project requirements (`>=3.11`)

### For All PRs

- [ ] `uv run ruff check` passes
- [ ] `uv run ruff format --check` passes
- [ ] `uv run ty check` passes
- [ ] `uv run pytest` passes

## Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_bicubic_gen_payload.py

# Run with verbose output
uv run pytest -v
```

## Code Style

- Line length: 100 characters
- Formatter: ruff format
- Linter: ruff (E, F, W, I, UP, B, SIM rules)
- Type checker: ty
