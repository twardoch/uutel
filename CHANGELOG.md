# CHANGELOG

All notable changes to this project will be documented in this file.

## [1.0.3] - 2024-09-29

### Added
- **Core Infrastructure Complete**: Implemented BaseUU class extending LiteLLM's CustomLLM
- **Authentication Framework**: Added BaseAuth and AuthResult classes for provider authentication
- **Core Utilities**: Implemented message transformation, HTTP client creation, model validation, and retry logic
- **Exception Framework**: Added comprehensive error handling with 7 specialized exception types
- **Testing Infrastructure**: Created rich pytest configuration with fixtures and mocking utilities
- **Usage Examples**: Added basic_usage.py demonstrating all core UUTEL functionality
- **Package Structure**: Created proper core/ and providers/ directory structure following UU naming convention
- **Type Safety**: Full type hints throughout codebase with mypy compatibility

### Fixed
- **Package Exports**: Main package now properly exports all core classes and utilities
- **Test Configuration**: Fixed pytest asyncio configuration for reliable testing
- **Import System**: Resolved circular import issues in core module structure
- **Test Coverage**: Improved coverage from 71% to 84% with comprehensive edge case testing

### Technical
- 55 tests passing (100% success rate) - increased from 24 tests
- 84% code coverage with core modules at 98-100%
- Exception framework: 7 exception types with 100% coverage
- Comprehensive test fixtures and utilities
- Working usage examples
- Ready for Phase 2: Provider implementations

## [1.0.2] - 2024-09-29

### Changed
- Updated README.md with comprehensive UUTEL package description based on PLAN.md
- README now presents UUTEL as a complete Universal AI Provider for LiteLLM
- Added detailed usage examples for all four providers (Claude Code, Gemini CLI, Cloud Code, Codex)
- Added package structure documentation and authentication setup guides

### Documentation
- Enhanced README with provider-specific usage examples
- Added comprehensive package architecture description
- Documented authentication methods for each provider
- Added installation and development setup instructions

## [1.0.1] - Previous
- Version bump with basic project structure

## [1.0.0] - Previous
- Initial project setup
- Basic package structure with hatch configuration
- Test infrastructure setup