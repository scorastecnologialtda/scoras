# Contributing to Scoras

Thank you for your interest in contributing to Scoras! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by the Scoras Code of Conduct:

- Be respectful and inclusive
- Be patient and welcoming
- Be thoughtful
- Be collaborative
- When disagreeing, try to understand why

## How to Contribute

### Reporting Bugs

If you find a bug in Scoras, please report it by creating an issue on the GitHub repository. When filing a bug report, please include:

1. A clear and descriptive title
2. A detailed description of the issue
3. Steps to reproduce the bug
4. Expected behavior
5. Actual behavior
6. Environment information (OS, Python version, etc.)
7. Any relevant logs or error messages

### Suggesting Enhancements

We welcome suggestions for enhancements to Scoras. To suggest an enhancement:

1. Create an issue on the GitHub repository
2. Use a clear and descriptive title
3. Provide a detailed description of the suggested enhancement
4. Explain why this enhancement would be useful
5. If possible, provide examples of how the enhancement would work

### Pull Requests

We actively welcome pull requests:

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Make your changes
4. Add or update tests as necessary
5. Ensure all tests pass
6. Update documentation as needed
7. Submit a pull request

#### Pull Request Guidelines

- Follow the existing code style
- Include tests for new features or bug fixes
- Update documentation for any changed functionality
- Keep pull requests focused on a single change
- Link the pull request to any related issues

## Development Setup

To set up Scoras for local development:

```bash
# Clone the repository
git clone https://github.com/scoras/scoras.git
cd scoras

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Testing

Scoras uses pytest for testing. To run the tests:

```bash
pytest
```

To run tests with coverage:

```bash
pytest --cov=scoras
```

## Documentation

Documentation is written in Markdown and built using MkDocs. To build the documentation locally:

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build and serve documentation
mkdocs serve
```

Then visit `http://localhost:8000` to view the documentation.

## Versioning

Scoras follows [Semantic Versioning](https://semver.org/). In short:

- MAJOR version for incompatible API changes
- MINOR version for added functionality in a backward-compatible manner
- PATCH version for backward-compatible bug fixes

## License

By contributing to Scoras, you agree that your contributions will be licensed under the project's MIT License.
