# Contributing to Truffle Occurrence Data Downloader

We welcome contributions to the Truffle Occurrence Data Downloader! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/truffle-occurrence-downloader.git
   cd truffle-occurrence-downloader
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install development dependencies**:
   ```bash
   pip install -e .[dev,geospatial]
   ```

5. **Run tests** to ensure everything works:
   ```bash
   pytest
   ```

## 📝 Types of Contributions

We welcome several types of contributions:

### 🐛 Bug Reports
- Use the GitHub issue tracker
- Include a clear description of the bug
- Provide steps to reproduce
- Include system information (OS, Python version, etc.)

### ✨ Feature Requests
- Use the GitHub issue tracker
- Describe the feature and its use case
- Consider if it fits the project's scope

### 🔧 Code Contributions
- Bug fixes
- New features
- Performance improvements
- Documentation updates
- Test improvements

## 🛠️ Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes
- Write clean, readable code
- Add tests for new functionality
- Update documentation as needed
- Follow the existing code style

### 3. Run Tests and Linting
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=truffle_downloader

# Check code style
black --check src/ tests/
flake8 src/ tests/

# Type checking
mypy src/
```

### 4. Commit Your Changes
```bash
git add .
git commit -m "Add: brief description of changes"
```

Use conventional commit messages:
- `Add:` for new features
- `Fix:` for bug fixes
- `Update:` for improvements
- `Remove:` for deletions
- `Docs:` for documentation changes

### 5. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## 📋 Code Style Guidelines

### Python Code
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Keep functions focused and small
- Use meaningful variable and function names

### Documentation
- Use clear, concise language
- Include examples in docstrings
- Update README.md for significant changes
- Add docstrings for new functions/classes

### Testing
- Write tests for new functionality
- Aim for good test coverage
- Use descriptive test names
- Test both success and failure cases

## 🧪 Testing Guidelines

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_downloader.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=truffle_downloader --cov-report=html
```

### Writing Tests
- Place tests in the `tests/` directory
- Use descriptive test names
- Test both positive and negative cases
- Mock external API calls
- Use fixtures for common test data

### Example Test Structure
```python
def test_function_name_success():
    """Test successful case"""
    # Arrange
    input_data = "test"
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected_output

def test_function_name_error():
    """Test error case"""
    # Arrange
    invalid_input = None
    
    # Act & Assert
    with pytest.raises(ValueError):
        function_under_test(invalid_input)
```

## 📚 Documentation Guidelines

### Code Documentation
- Use Google-style docstrings
- Include type hints
- Document parameters and return values
- Include usage examples

### Example Docstring
```python
def download_species(self, species: List[str], **kwargs) -> pd.DataFrame:
    """
    Download occurrence data for specified truffle species.
    
    Args:
        species: List of species names to download
        **kwargs: Additional filtering parameters
        
    Returns:
        DataFrame with occurrence records
        
    Example:
        >>> downloader = GBIFTruffleDownloader()
        >>> data = downloader.download_species(["Tuber melanosporum"])
        >>> print(len(data))
        150
    """
```

### README Updates
- Update installation instructions if dependencies change
- Add new features to the features list
- Update examples if API changes
- Keep the quick start section current

## 🔍 Code Review Process

### For Contributors
- Ensure all tests pass
- Address review feedback promptly
- Keep pull requests focused and small
- Respond to comments constructively

### For Reviewers
- Be constructive and helpful
- Focus on code quality and correctness
- Check that tests are adequate
- Verify documentation is updated

## 🐛 Reporting Issues

### Before Reporting
- Check existing issues
- Ensure you're using the latest version
- Try to reproduce the issue

### Issue Template
```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**
- OS: [e.g. Ubuntu 20.04]
- Python version: [e.g. 3.9]
- Package version: [e.g. 1.0.0]

**Additional context**
Any other context about the problem.
```

## 🎯 Feature Request Guidelines

### Before Requesting
- Check existing feature requests
- Consider if it fits the project scope
- Think about implementation complexity

### Feature Request Template
```markdown
**Is your feature request related to a problem?**
A clear description of what the problem is.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Any alternative solutions or workarounds.

**Additional context**
Any other context about the feature request.
```

## 📦 Release Process

### Version Numbering
We use semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)

### Release Checklist
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Version number is updated
- [ ] CHANGELOG.md is updated
- [ ] Release notes are written

## 🤝 Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow the golden rule

### Getting Help
- Check the documentation first
- Search existing issues
- Ask questions in discussions
- Be specific about your problem

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/truffle-research/truffle-occurrence-downloader/issues)
- **Discussions**: [GitHub Discussions](https://github.com/truffle-research/truffle-occurrence-downloader/discussions)
- **Email**: support@truffle-downloader.org

## 🙏 Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to the Truffle Occurrence Data Downloader! 🍄