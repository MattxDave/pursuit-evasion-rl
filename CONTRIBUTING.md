# Contributing to Pursuit-Evasion Hybrid RL

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/pursuit-evasion-rl.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes
6. Commit: `git commit -m "Add your descriptive commit message"`
7. Push: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run a quick test
python train.py --preset fast
python simulate.py
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to classes and functions
- Keep functions focused and modular

## Testing

Before submitting a PR:
1. Ensure all imports work correctly
2. Test training with `--preset fast`
3. Test simulation with default parameters
4. Verify multi-agent mode if you made changes there

## Reporting Issues

When reporting bugs, please include:
- Python version
- OS and version
- Steps to reproduce
- Expected vs actual behavior
- Relevant error messages or logs

## Feature Requests

We welcome feature suggestions! Please open an issue describing:
- The problem you're trying to solve
- Your proposed solution
- Any alternative solutions you've considered

## Pull Request Guidelines

- Keep PRs focused on a single feature or bug fix
- Update documentation if needed
- Add comments for complex logic
- Reference related issues

## Questions?

Feel free to open an issue for questions or discussions!
