# Contributing to Facial Authentication System

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- Clear and descriptive title
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or screenshots

### Suggesting Features

Feature suggestions should include:

- Clear use case description
- Proposed solution
- Benefits to the project
- Alternatives considered

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**
4. **Run tests**: `pytest tests/`
   - If globally installed plugins interfere, run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/` instead.
5. **Run linters**: `black app/` and `flake8 app/`
6. **Commit changes**: Follow our commit message conventions
7. **Push to your fork**: `git push origin feature/your-feature-name`
8. **Create a Pull Request**

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(api): add temporal liveness endpoint
fix(models): correct face detection edge case
docs(readme): update installation instructions
```

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/facial-auth-system.git
cd facial-auth-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

5. Run the application:
```bash
uvicorn app.main:app --reload
```

### Code Style

We use:
- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking
- **Isort** for import sorting

Run all checks:
```bash
black app/ training/
flake8 app/ training/
mypy app/
isort app/ training/
```

### Testing

We require tests for all new features:

1. Unit tests for individual components
2. Integration tests for workflows
3. Update existing tests if APIs change

Run tests:
```bash
# All tests
pytest

# With coverage
pytest --cov=app tests/

# Specific test file
pytest tests/test_models.py

# Verbose output
pytest -v
```

### Documentation

- Update README.md for user-facing changes
- Add docstrings to all functions and classes
- Update API documentation if endpoints change
- Include examples in docstrings

### Project Structure

```
facial-auth-system/
├── app/                    # Main application code
│   ├── models/            # ML models
│   ├── services/          # Business logic
│   ├── api/              # API endpoints
│   ├── core/             # Core utilities
│   └── utils/            # Helper functions
├── frontend/             # React application
├── training/             # Training scripts
├── tests/               # Test suite
├── deployment/          # Deployment configs
└── docs/               # Documentation
```

### Areas for Contribution

- ML model improvements
- API enhancements
- Frontend features
- Performance optimizations
- Documentation
- Bug fixes
- Test coverage
- Deployment automation

### Review Process

1. All PRs require at least one review
2. CI checks must pass
3. Code coverage should not decrease
4. Follow code style guidelines

### Questions?

Feel free to open an issue for questions or start a discussion in Discussions tab.

Thank you for contributing! 🎉

