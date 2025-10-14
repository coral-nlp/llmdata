# Contributing to LLMData

## Development Setup

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/llmdata.git
   cd llmdata
   ```

2. **Set up development environment**:
   ```bash
   make dev-setup
   ```

   This will install the package with dev dependencies and set up pre-commit hooks.

### Development Tools

#### Pre-commit Hooks

Pre-commit hooks automatically run before each commit to ensure code quality:

- **Formatting**: `black` and `ruff format` for consistent code formatting
- **Import sorting**: `isort` for organized imports
- **Linting**: `ruff` for fast, comprehensive linting (replaces flake8)
- **Type checking**: `mypy` for static type analysis
- **Security**: `bandit` for security vulnerability scanning
- **Documentation**: `pydocstyle` for docstring standards
- **General**: Various hooks for trailing whitespace, file endings, YAML/JSON validation

#### Available Commands

Use the Makefile for development tasks:

```bash
# Setup
make dev-setup              # Install dev dependencies and pre-commit hooks

# Code Quality
make lint                   # Run linting (ruff)
make format                 # Format code (isort, ruff-format)
make type-check             # Run type checking (mypy)
make security-check         # Run security checks (bandit)
make doc-check              # Check documentation style
make check                  # Run all checks

# Testing
make test                   # Run tests
make test-cov               # Run tests with coverage

# Development Workflow
make fix                    # Fix auto-fixable issues
make dev-check              # Full development check (fix + all checks + tests)

# Cleanup
make clean                  # Remove build artifacts
```

### Code Style Guidelines

#### Python Style

- **Line length**: 120 characters
- **Imports**: Sorted
- **Type hints**: Required for all public functions and methods
- **Docstrings**: Google style for all public functions, classes, and modules

#### Pydantic Models

All processor and aggregator classes should use Pydantic field annotations and register themselves:

```python
from pydantic import Field
from llmdata.core.ops import MapFn
from llmdata.core.registry import components


@components.add("processor", "example")
class ExampleProcessor(MapFn):
   name: str = Field(default="example", description="Processor name")
   on: str = Field(default="text", description="Input column")
   to: str = Field(default="result", description="Output column")
   param: int = Field(default=10, description="Example parameter", gt=0)
```

### Testing

Run tests before submitting:

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test
uv run pytest tests/test_specific.py -v
```

### Continuous Integration

The pre-commit hooks ensure that:

1. Code is properly formatted
2. Imports are sorted
3. Type annotations are correct
4. No security vulnerabilities are introduced
5. Documentation follows standards
6. No syntax errors exist

### Troubleshooting

#### Pre-commit Issues

If pre-commit hooks fail:

1. **Fix automatically**: Run `make fix` to auto-fix most issues
2. **Manual fixes**: Address remaining issues based on the error messages
3. **Skip hooks** (only if necessary): `git commit --no-verify`

#### Type Checking Errors

- Add type annotations for missing types
- Use `# type: ignore` comments sparingly for external library issues
- Update `mypy` configuration in `pyproject.toml` if needed

## Submitting Changes

1. **Create a feature branch**: `git checkout -b feature/your-feature`
2. **Make changes**: Edit code following the guidelines
3. **Run checks**: `make dev-check` to ensure all checks pass
4. **Commit changes**: Pre-commit hooks will run automatically
5. **Push and create PR**: Submit a pull request for review

## Additional Resources

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
