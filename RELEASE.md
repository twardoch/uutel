# UUTEL Release Process

This document describes the automated release process for UUTEL.

## Overview

UUTEL uses a comprehensive, automated release management system with multiple workflows:

1. **Semantic Release** - Automated versioning based on conventional commits
2. **Manual Release Preparation** - For planned releases with full control
3. **Release Validation** - Comprehensive pre-release checks
4. **PyPI Publishing** - Automated publication to PyPI

## Release Workflows

### 1. Automated Semantic Release

**Trigger**: Push to `main` branch with conventional commits
**Workflow**: `.github/workflows/semantic-release.yml`

The semantic release workflow automatically:
- Analyzes commits since last release
- Determines version bump type (major/minor/patch)
- Runs comprehensive validation
- Creates version bump commit
- Creates and pushes git tag
- Triggers PyPI publication

#### Conventional Commit Format

Use these commit prefixes to trigger automatic releases:

- `feat:` - New feature (triggers **minor** version bump)
- `fix:` - Bug fix (triggers **patch** version bump)
- `perf:` - Performance improvement (triggers **patch** version bump)
- `BREAKING CHANGE:` - Breaking change (triggers **major** version bump)

**Examples:**
```bash
git commit -m "feat: add new authentication provider"
git commit -m "fix: resolve connection timeout issue"
git commit -m "feat!: redesign API interface

BREAKING CHANGE: The API interface has been completely redesigned"
```

#### Version Bump Rules

| Commit Type | Version Bump | Example |
|-------------|--------------|---------|
| `BREAKING CHANGE:` | Major (1.0.0 → 2.0.0) | API changes |
| `feat:` | Minor (1.0.0 → 1.1.0) | New features |
| `fix:`, `perf:` | Patch (1.0.0 → 1.0.1) | Bug fixes |

### 2. Manual Release Preparation

**Trigger**: Manual workflow dispatch
**Workflow**: `.github/workflows/release-preparation.yml`

For planned releases with full control:

1. Go to **Actions** → **Prepare Release**
2. Select version type: `patch`, `minor`, `major`, or `prerelease`
3. Optionally run in dry-run mode first
4. Workflow creates a release branch and PR

#### Process:
1. Validates package readiness
2. Calculates next version
3. Updates version files
4. Generates changelog
5. Creates release branch
6. Opens pull request for review

### 3. PyPI Publication

**Trigger**: Git tag creation (format: `v*`)
**Workflow**: `.github/workflows/release.yml`

Automatically triggered when a version tag is pushed:

1. **Pre-release validation**:
   - Distribution readiness check
   - Production health validation
   - Package integrity verification

2. **Build process**:
   - Creates wheel and source distributions
   - Validates with `twine check`

3. **Publication**:
   - Publishes to PyPI
   - Creates GitHub release with artifacts

## Pre-Release Validation

Both automated and manual releases include comprehensive validation:

### Distribution Validation
- ✅ PyPI package structure
- ✅ Metadata completeness
- ✅ Build configuration
- ✅ Required files present
- ✅ Version format compliance

### Production Readiness
- ✅ Python version compatibility
- ✅ Core dependencies available
- ✅ Package integrity verification
- ✅ System compatibility
- ✅ Runtime environment validation

### Quality Gates
- ✅ All tests passing
- ✅ Code coverage > 90%
- ✅ No security vulnerabilities
- ✅ Linting checks pass
- ✅ Type checking passes

## Release Types

### Patch Release (1.0.0 → 1.0.1)
- Bug fixes
- Performance improvements
- Documentation updates
- Internal refactoring

### Minor Release (1.0.0 → 1.1.0)
- New features
- New functionality
- Backwards-compatible API additions

### Major Release (1.0.0 → 2.0.0)
- Breaking changes
- API redesigns
- Backwards-incompatible changes

### Pre-release (1.0.0 → 1.1.0-rc1)
- Release candidates
- Beta releases
- Testing versions

## Manual Release Process

If you need to create a release manually:

### Option 1: Use Semantic Release (Recommended)

1. **Make conventional commits**:
   ```bash
   git commit -m "feat: add amazing new feature"
   git push origin main
   ```

2. **Automatic process**:
   - Semantic release workflow detects commits
   - Creates version bump and tag
   - Publishes to PyPI automatically

### Option 2: Use Release Preparation Workflow

1. **Run preparation workflow**:
   - Go to Actions → Prepare Release
   - Select version type
   - Review and merge the created PR

2. **Create tag manually**:
   ```bash
   git tag v1.2.3
   git push origin v1.2.3
   ```

### Option 3: Completely Manual

1. **Update version**:
   ```bash
   # Update src/uutel/_version.py
   __version__ = "1.2.3"
   ```

2. **Update changelog**:
   ```bash
   # Add entry to CHANGELOG.md
   ```

3. **Commit and tag**:
   ```bash
   git add src/uutel/_version.py CHANGELOG.md
   git commit -m "chore: release v1.2.3"
   git tag v1.2.3
   git push origin main
   git push origin v1.2.3
   ```

## Environment Setup

### Required Secrets

Ensure these secrets are configured in the repository:

- `PYPI_TOKEN` - PyPI API token for package publishing
- `GITHUB_TOKEN` - Automatically provided by GitHub

### Branch Protection

Main branch should have protection rules:
- Require pull request reviews
- Require status checks to pass
- Require branches to be up to date
- Include administrators in restrictions

## Troubleshooting

### Common Issues

#### 1. Release Validation Fails
```bash
❌ Package is not ready for PyPI release
```
**Solution**: Run local validation and fix issues:
```python
from uutel.core.distribution import get_distribution_summary
summary = get_distribution_summary()
print(summary)
```

#### 2. Version Already Exists
```bash
❌ Version 1.2.3 already exists on PyPI
```
**Solution**: Increment version number and retry

#### 3. Build Fails
```bash
❌ Wheel file missing
```
**Solution**: Check `pyproject.toml` configuration and build dependencies

### Debug Commands

**Check current version**:
```python
from uutel import __version__
print(__version__)
```

**Validate distribution**:
```python
from uutel.core.distribution import validate_pypi_readiness
print(validate_pypi_readiness())
```

**Check health**:
```python
from uutel.core.health import get_health_summary
print(get_health_summary())
```

## Best Practices

1. **Use conventional commits** for automatic versioning
2. **Test releases** with pre-release versions first
3. **Update documentation** before major releases
4. **Review changelogs** generated automatically
5. **Monitor PyPI** publication for issues
6. **Tag important releases** for easy reference

## Monitoring

### Release Dashboard

Monitor releases at:
- **GitHub Releases**: https://github.com/YOUR_ORG/uutel/releases
- **PyPI Package**: https://pypi.org/project/uutel/
- **GitHub Actions**: https://github.com/YOUR_ORG/uutel/actions

### Metrics to Track

- Release frequency
- Time from commit to publication
- Validation failure rate
- Download statistics from PyPI
- User adoption of new versions