# UUTEL Dependencies

This document lists all dependencies used by UUTEL and the rationale for including each.

## Core Dependencies

### LiteLLM Integration
- **litellm>=1.70.0**: Core requirement for extending LiteLLM's provider ecosystem
  - Provides `CustomLLM` base class that `BaseUU` extends
  - Enables seamless integration with LiteLLM's unified interface
  - Required for model registration and routing

### HTTP and Async Support
- **httpx>=0.25.0**: Modern async-capable HTTP client
  - Used for provider API communications
  - Supports both sync and async operations
  - Better type hints and modern Python support than requests
  - Required for reliable connection management

- **aiohttp>=3.8.0**: Async HTTP server/client framework
  - Used for async HTTP operations where httpx is insufficient
  - Provides WebSocket support for streaming responses
  - Required for advanced async communication patterns

### Data Validation and Parsing
- **pydantic>=2.0.0**: Data validation and settings management
  - Ensures type safety for request/response models
  - Provides runtime validation of provider responses
  - Modern Pydantic v2 for performance and features

- **pydantic-settings>=2.0.0**: Configuration management with Pydantic
  - Handles environment variable loading
  - Provides structured configuration validation
  - Integrates seamlessly with main Pydantic models

### Logging and Monitoring
- **loguru>=0.7.0**: Modern Python logging library
  - Simpler and more powerful than standard logging
  - Better structured logging support
  - Excellent for debugging provider interactions

## Authentication Dependencies

### Google Cloud and OAuth
- **google-auth>=2.15.0**: Google authentication library
  - Required for Google Cloud Code and Gemini CLI providers
  - Handles service account and OAuth flows
  - Industry standard for Google API authentication

- **google-auth-oauthlib>=1.0.0**: OAuth 2.0 flows for Google APIs
  - Required for interactive OAuth flows
  - Supports PKCE for secure OAuth in Claude Code provider
  - Handles token refresh and storage

- **google-cloud-core>=2.0.0**: Core Google Cloud client library
  - Provides common utilities for Google Cloud services
  - Required for Cloud Code provider implementation
  - Ensures consistent error handling across Google services

## CLI and User Interface Dependencies

### Command Line Interface
- **typer>=0.9.0**: Modern CLI library built on Click
  - Creates rich command line interfaces
  - Automatic help generation with type hints
  - Better than argparse for complex CLI tools

- **rich>=13.0.0**: Rich text and beautiful formatting
  - Provides colored output and progress bars
  - Enhances CLI user experience
  - Used for formatted error messages and status output

## Development and Testing Dependencies

### Testing Framework
- **pytest>=8.3.4**: Modern Python testing framework
  - Comprehensive test suite support
  - Better fixtures and parametrization than unittest
  - Industry standard for Python testing

- **pytest-cov>=6.0.0**: Coverage reporting for pytest
  - Measures test coverage across codebase
  - Ensures quality through coverage metrics
  - Required for comprehensive testing

- **pytest-xdist>=3.6.1**: Distributed testing for pytest
  - Enables parallel test execution
  - Faster test runs for large test suites
  - Improves development workflow

- **pytest-asyncio>=0.25.3**: Async testing support
  - Required for testing async provider methods
  - Handles event loops properly in tests
  - Essential for async/await test cases

- **pytest-mock>=3.15.0**: Mocking framework for pytest
  - Provides mocking capabilities for external APIs
  - Enables isolated unit testing
  - Required for testing without hitting real APIs

- **coverage[toml]>=7.6.12**: Coverage measurement tool
  - Tracks test coverage metrics
  - TOML configuration support
  - Required for coverage reporting

### Code Quality Tools
- **ruff>=0.9.7**: Fast Python linter and formatter
  - Modern replacement for flake8, black, isort
  - Extremely fast linting and formatting
  - Single tool for multiple code quality checks

- **mypy>=1.15.0**: Static type checker
  - Ensures type safety throughout codebase
  - Catches type-related bugs before runtime
  - Required for maintaining type hints quality

- **pre-commit>=4.1.0**: Git pre-commit hooks framework
  - Ensures code quality before commits
  - Automatically runs linting and formatting
  - Prevents broken code from entering repository

- **bandit>=1.8.0**: Security linter for Python code
  - Scans code for common security issues
  - Identifies potential vulnerabilities
  - Required for maintaining security standards

- **safety>=4.0.0**: Dependency vulnerability scanner
  - Checks dependencies for known security vulnerabilities
  - Provides security alerts for outdated packages
  - Required for maintaining secure dependencies

### Import Organization
- **isort**: Python import sorting tool
  - Automatically sorts and organizes imports
  - Configured for Black compatibility
  - Maintains consistent import style across codebase

## Package Selection Rationale

### Why These Specific Versions?
- **Minimum versions**: Chosen to ensure compatibility with older Python installations
- **Modern libraries**: Selected for better type hints, async support, and performance
- **Stable APIs**: All dependencies have stable, well-documented APIs
- **Active maintenance**: All packages are actively maintained with regular updates

### Alternative Considerations
- **requests vs httpx**: httpx chosen for async support and modern Python compatibility
- **argparse vs typer**: typer chosen for better type hints and automatic documentation
- **unittest vs pytest**: pytest chosen for better fixtures and more readable tests
- **black vs ruff**: ruff chosen for speed and comprehensive functionality

## Security Considerations
- All dependencies are from trusted sources (PyPI verified packages)
- Version pinning prevents unexpected updates that could introduce vulnerabilities
- Regular dependency updates planned to address security issues
- No dependencies with known security vulnerabilities

## Future Dependency Plans
- Consider adding `tenacity` for more sophisticated retry logic
- May add `click-completion` for shell completions
- Consider `structlog` for more structured logging
- Evaluate `pydantic-extra-types` for specialized validation types
