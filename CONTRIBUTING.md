# Contributing to UUTEL

Welcome to UUTEL! We appreciate your interest in contributing to the Universal AI Provider for LiteLLM. This guide will help you get started with development, testing, and contributing to the project.

## Quick Start for Contributors

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/uutel.git
   cd uutel
   ```

2. **Set Up Development Environment**
   ```bash
   # Install uv if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Set up project dependencies
   uv sync --all-extras

   # Alternatively, use make for guided setup
   make setup
   ```

3. **Verify Installation**
   ```bash
   # Run tests to ensure everything works
   uvx hatch run test

   # Or use make
   make test
   ```

## Development Environment

### Prerequisites

- **Python 3.10+**: UUTEL supports Python 3.10, 3.11, and 3.12
- **uv**: Modern Python package manager (recommended)
- **git**: Version control

### Development Tools

UUTEL uses modern Python tooling for an excellent developer experience:

- **[Hatch](https://hatch.pypa.io/)**: Project management and virtual environments
- **[uv](https://github.com/astral-sh/uv)**: Fast Python package installer and resolver
- **[Ruff](https://github.com/astral-sh/ruff)**: Lightning-fast linting and formatting
- **[MyPy](https://mypy.readthedocs.io/)**: Static type checking
- **[Pytest](https://pytest.org/)**: Testing framework with async support
- **[Pre-commit](https://pre-commit.com/)**: Git hooks for code quality

### Environment Setup

#### Option 1: Using Hatch (Recommended)
```bash
# Install hatch if not already installed
pip install hatch

# Create and activate development environment
hatch shell

# Run tests
hatch run test

# Run tests with coverage
hatch run test-cov

# Run linting
hatch run lint

# Format code
hatch run format

# Type checking
hatch run typecheck
```

#### Option 2: Using uv
```bash
# Install all dependencies including dev tools
uv sync --all-extras

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/uutel --cov-report=term-missing

# Lint code
uv run ruff check src/uutel tests

# Format code
uv run ruff format src/uutel tests

# Type check
uv run mypy src/uutel tests
```

#### Option 3: Using Make Commands
```bash
# See all available commands
make help

# Set up development environment
make setup

# Run all checks
make check

# Run tests
make test

# Run tests with coverage
make test-cov

# Format and lint
make format
make lint

# Type checking
make typecheck

# Security scanning
make security

# Run everything (CI-like)
make ci
```

## Code Standards

### Code Quality Requirements

- **Test Coverage**: Maintain >90% test coverage
- **Type Hints**: All functions must have type hints
- **Docstrings**: All public functions require docstrings
- **Function Length**: Keep functions under 20 lines
- **File Length**: Keep files under 200 lines
- **No Complexity**: Avoid deep nesting (max 3 levels)

### Naming Conventions

UUTEL follows the **Universal Unit (UU)** naming pattern:

- **Provider Classes**: `{ProviderName}UU` (e.g., `ClaudeCodeUU`, `GeminiCLIUU`)
- **Authentication Classes**: `{ProviderName}Auth` (e.g., `ClaudeCodeAuth`)
- **Transform Classes**: `{ProviderName}Transform` (e.g., `GeminiCLITransform`)
- **Model Classes**: `{ProviderName}Models` (e.g., `CodexModels`)

### Code Style

- **Line Length**: 88 characters (configured in ruff)
- **Import Ordering**: isort compatible, uutel as first-party
- **String Quotes**: Prefer double quotes
- **Type Hints**: Use modern syntax (`list[str]`, `dict[str, Any]`, `str | None`)

### Pre-commit Hooks

Set up pre-commit hooks to ensure code quality:

```bash
# Install pre-commit
uv add --dev pre-commit

# Install hooks
pre-commit install

# Run hooks on all files
pre-commit run --all-files
```

## Testing Guidelines

### Test Structure

- **Location**: All tests in `tests/` directory
- **Naming**: Test files start with `test_`, functions start with `test_`
- **Organization**: Mirror source structure (`src/module.py` → `tests/test_module.py`)

### Test Types

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Ensure speed requirements
4. **Security Tests**: Validate security practices

### Writing Tests

#### Test Naming Convention
```python
def test_function_name_when_condition_then_expected_result():
    """Test function behavior under specific conditions."""
    pass
```

#### Test Structure
```python
def test_validate_tool_schema_when_valid_schema_then_returns_true():
    """Test that validate_tool_schema returns True for valid OpenAI tool schema."""
    # Arrange
    valid_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information"
        }
    }

    # Act
    result = validate_tool_schema(valid_tool)

    # Assert
    assert result is True, "Valid tool schema should return True"
```

#### Async Tests
```python
import pytest

@pytest.mark.asyncio
async def test_async_function_when_called_then_succeeds():
    """Test async function behavior."""
    result = await some_async_function()
    assert result is not None
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run tests in parallel (faster, but may cause some test failures)
make test-fast

# Run specific test file
uvx hatch run test tests/test_specific_module.py

# Run with verbose output
uvx hatch run test -v

# Run specific test
uvx hatch run test tests/test_utils.py::test_specific_function

# Run async tests specifically
uvx hatch run test -k "async"

# Run integration tests
uvx hatch run test -m integration
```

### Test Markers

- `@pytest.mark.asyncio`: For async tests
- `@pytest.mark.slow`: For slow-running tests
- `@pytest.mark.integration`: For integration tests

## Development Workflow

### Test-Driven Development (TDD)

UUTEL follows TDD principles:

1. **RED**: Write a failing test
2. **GREEN**: Write minimal code to pass
3. **REFACTOR**: Clean up while keeping tests green
4. **REPEAT**: Next feature

### Development Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write Tests First**
   ```bash
   # Write failing test
   echo "def test_new_feature(): assert False" >> tests/test_new_module.py

   # Verify it fails
   make test
   ```

3. **Implement Feature**
   ```bash
   # Write minimal code to pass test
   # Edit source files

   # Verify tests pass
   make test
   ```

4. **Ensure Quality**
   ```bash
   # Run all quality checks
   make check

   # Run full CI-like checks
   make ci
   ```

5. **Commit Changes**
   ```bash
   # Stage changes
   git add .

   # Use conventional commits
   git commit -m "feat: add new feature description"
   ```

### Conventional Commits

Use conventional commit messages for automatic versioning:

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `test:` - Test additions/changes
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `chore:` - Maintenance tasks

Examples:
```bash
git commit -m "feat: add streaming support for Claude Code provider"
git commit -m "fix: handle network timeout in authentication"
git commit -m "docs: update README with new provider examples"
git commit -m "test: add edge case tests for tool validation"
```

## Pull Request Guidelines

### Before Submitting

1. **All Tests Pass**
   ```bash
   make ci
   ```

2. **Code Coverage Maintained**
   ```bash
   make test-cov
   # Ensure coverage stays above 90%
   ```

3. **Code Quality Checks**
   ```bash
   make check
   make typecheck
   make security
   ```

4. **Documentation Updated**
   - Update relevant docstrings
   - Add examples if needed
   - Update CHANGELOG.md if significant

### PR Requirements

- **Clear Description**: Explain what and why
- **Tests Included**: Cover new functionality
- **No Breaking Changes**: Unless explicitly discussed
- **Clean History**: Squash commits if needed
- **Conventional Commits**: Use proper commit messages

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass
- [ ] New tests added
- [ ] Coverage maintained >90%

## Checklist
- [ ] Code follows project conventions
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes
```

## Architecture Guidelines

### Project Structure

```
uutel/
├── src/uutel/
│   ├── __init__.py           # Main exports
│   ├── core/                 # Core infrastructure
│   │   ├── base.py          # BaseUU class
│   │   ├── auth.py          # Authentication framework
│   │   ├── exceptions.py    # Custom exceptions
│   │   └── utils.py         # Utilities
│   └── providers/           # Provider implementations (future)
├── tests/                   # Test suite
├── examples/                # Usage examples
└── docs/                    # Documentation
```

### Design Patterns

1. **Universal Unit Pattern**: All providers inherit from `BaseUU`
2. **Composition over Inheritance**: Use composition for complex behaviors
3. **Dependency Injection**: Accept dependencies as parameters
4. **Fail Fast**: Validate early, handle errors gracefully

### Provider Implementation Guidelines

When implementing new providers:

1. **Follow UU Pattern**: Extend `BaseUU`
2. **Implement Required Methods**: `completion()`, `stream_completion()`
3. **Add Authentication**: Create `{Provider}Auth` class
4. **Message Transformation**: Implement format conversion
5. **Error Handling**: Map provider errors to UUTEL exceptions
6. **Tests**: Comprehensive test coverage

## Common Tasks

### Adding a New Utility Function

1. **Write Test First**
   ```python
   # tests/test_utils.py
   def test_new_utility_when_called_then_returns_expected():
       result = new_utility("input")
       assert result == "expected"
   ```

2. **Implement Function**
   ```python
   # src/uutel/core/utils.py
   def new_utility(input_value: str) -> str:
       """Brief description of what this does."""
       return process(input_value)
   ```

3. **Add to Exports**
   ```python
   # src/uutel/__init__.py
   from .core.utils import new_utility
   ```

### Debugging Tests

```bash
# Run with debugger
uvx hatch run test --pdb

# Run with verbose output
uvx hatch run test -v -s

# Run specific failing test
uvx hatch run test tests/test_module.py::test_failing_function -v

# Check test coverage details
uvx hatch run test-cov --cov-report=html
open htmlcov/index.html
```

### Performance Testing

```bash
# Run performance tests
uvx hatch run test -m slow

# Profile specific function
python -m cProfile -s cumulative examples/performance_test.py
```

## Getting Help

### Resources

- **Documentation**: Check README.md and ARCHITECTURE.md
- **Examples**: Review files in `examples/` directory
- **Tests**: Look at test files for usage patterns
- **Issues**: Search existing GitHub issues

### Communication

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Pull Request Reviews**: For code feedback

### Development Questions

If you're stuck:

1. Check existing tests for similar patterns
2. Review the architecture documentation
3. Look at examples in the `examples/` directory
4. Search existing issues and discussions
5. Create a new issue with specific details

## Release Process

UUTEL uses automated semantic versioning:

1. **Conventional Commits**: Use proper commit messages
2. **Automatic Versioning**: Based on commit types
3. **Changelog Generation**: Automatically updated
4. **GitHub Releases**: Created automatically
5. **PyPI Publishing**: Automated on release

### Version Types

- `feat:` commits → Minor version bump (0.1.0 → 0.2.0)
- `fix:` commits → Patch version bump (0.1.0 → 0.1.1)
- `BREAKING CHANGE:` → Major version bump (0.1.0 → 1.0.0)

## Final Notes

### Key Principles

1. **Simplicity First**: Avoid over-engineering
2. **Test Everything**: Untested code is broken code
3. **Document Decisions**: Why, not just what
4. **Performance Matters**: Keep overhead minimal
5. **Security Conscious**: Handle credentials securely

### Quality Gates

Before any contribution:

- [ ] All tests pass (318+ tests)
- [ ] Coverage >90%
- [ ] No linting errors
- [ ] Type checking passes
- [ ] Security scan clean
- [ ] Documentation updated

Thank you for contributing to UUTEL! Your efforts help make AI provider integration simpler and more reliable for everyone.